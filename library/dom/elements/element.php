<?php
/**
 * This class is the common superclass of all DOM elements.
 *
 * 
 * @author durso
 */
namespace library\dom\elements;
use library\dom\object;
use library\utils;
use library\event\listener;
use library\event\event;
use library\mediator\nodeElement;



abstract class element extends object{
    
    private $listener = array();
    /**
     *
     * @var array All attributes and properties given to an element
     */
    protected $attributes = array();
    /**
     *
     * @var string html tag
     */
    protected $tag;
    /*
     * 
     * Add CSS class to element
     * @param string $className CSS class name to be added to the element
     * @return void
     */
    public function addClass($className){
        if(!array_key_exists("class", $this->attributes)){
            $this->attributes["class"] = array();
        }
        if(!in_array($className,$this->attributes["class"])){
            $this->attributes["class"][] = $className;
        }
        if($this->isRendered){
            $this->updateJS('addClass',$className);
        }
    }
    /*
     * 
     * Add CSS class to element
     * @param string $className CSS class name to be added to the element
     * @return void
     */
    public function removeClass($className){
        if($this->hasClass($className)){
            utils::array_remove($this->attributes["class"], $className);
            $this->updateJS('removeClass',$className);
        }
    }
    /*
     * 
     * Check if element has class
     * @return void
     */
    public function hasClass($className = false){
        if($className){
            if($this->hasClass()){
                return in_array($className,$this->attributes['class']);
            }
        }
        return (isset($this->attributes["class"]) && !empty($this->attributes["class"]));
    }
    public function getClassbyIndex($index){
        return $this->attributes["class"][$index];
    }
    /*
     * 
     * Set element attribute
     * @param string $attribute name of the attribute
     * @param string $value value of the attribute
     * @return void
     */
    public function attr($attribute,$value = false){
        if(!$value){
            if(isset($this->attribute[$attribute])){
                return $this->attribute[$attribute];
            }
            return false;
        }
        if($attribute != "class"){
            $this->attributes[$attribute] = $value; 
        }
        if($this->isRendered){
            $this->updateJS('attr',$value,$attribute);
        }
    }
    
    public function removeAttr($attribute){
        if(isset($this->attribute[$attribute])){
            unset($this->attribute[$attribute]);
        }
        if($this->isRendered){
            $this->updateJS('removeAttr',$attribute);
        }
    }
    /*
     * 
     * Render html tag attributes
     * @param mixed string or list of values to be added to the element
     * @param string $field name of the attribute
     * @return string|void
     */
    protected function renderAttributes($value, $field){
        if($value === true){
            return $field;
        }
        $attribute = ' '.$field.'="';
        if(is_array($value) && !empty($value)){
           return $attribute.implode(" ",array_unique($value)).'"';
        } else {
            if($value){
                return $attribute.$value.'"';
            }
        }
            
    }
    /*
     * 
     * Set CSS id
     * @param string $element html tag
     * @return void
     */
    public function setId($id,$force = false,$seed = 0){
        if($force){
           $this->attr('id',$id);
        } else {
            if(!$this->hasId()){
                $uid = $id."-".utils::randomGenerator();
                $this->attr('id',$uid);
            }
        }
    }
    /*
     * 
     * Get CSS id
     * @return void
     */
    public function getId(){
        return $this->attributes["id"];
    }
    /*
     * 
     * Check if element has id
     * @return void
     */
    public function hasId(){
        return (isset($this->attributes["id"]) && !empty($this->attributes["id"]));
    }
    /*
     * 
     * Get element tag
     * @return string
     */
    public function getTag(){
        return $this->tag;
    }
    protected function openTag(){
        $this->html = "<".$this->tag;
        foreach($this->attributes as $key => $value){
            $this->html .= $this->renderAttributes($value, $key);
        }
        $this->html .= ">";
    }
    /*
     * 
     * Get element unique identifier
     * @return string
     */
    public function getUid(){
        if($this->hasId()){
            return "#".$this->getId();
        }
        
        return $this->buildSelector();
        
    }
    

    /*
     * 
     * addEventListener an event and a function to the element
     * @param string $event the name of the event
     * @param callable $callback a function to be bound to the event
     * @return void
     */
    public function addEventListener($event, $callback,$args = array()){
        $this->setId($this->tag);
        $this->addClass($event);
        $this->listener[$event] = new listener($this,$event,$callback,$args);
    }
    public function removeEventListener($event){
        $this->removeClass($event);
        unset($this->listener[$event]);
    }
    
    public function fire(event $event){
        $this->listener[$event->getType()]->fire($event);
    }
    
    public function is($selector){
        if(preg_match("/(^([\w]+)\.([\w+\-*]+))/", $selector)){
            $comp = explode(".", $selector);
            return $this->tag == $comp[0] && $this->hasClass($comp[1]);
        } elseif(preg_match("/(^([\w]+)#([\w+\-*]+))/", $selector)){
            $comp = explode(".", $selector);
            return $this->tag == $comp[0] && $this->attributes['id'] == $comp[1];
        } elseif($selector[0] == '#'){
            return $this->attributes['id'] == $selector;
        } elseif($selector[0] == '.'){
            return $this->hasClass($selector);
        } else {
            return $this->tag == $selector;
        }
    }
    
    public function closest($selector){
        return nodeElement::closest($this,$selector);
    }
    
    /*
     * 
     * Get CSS selector for the element
     * @return string
     */
    protected function buildSelector(){
        $selector = nodeElement::buildSelector($this);
        $selector .= $this->tag;
        if($this->hasClass()){
            $selector .= ".".$this->getClassByIndex(0);
        }
        $selector .= $this->nthChild();
        return $selector;
    }
    /*
     * 
     * Get the nth-child selector of the element
     * @return string
     */
    public function nthChild(){
        $selector = "";
        $index = nodeElement::childIndex($this);
        if($index > -1){
            $index++;
            $selector = ":nth-child($index)";
        } 
        return $selector;
    }
    
    
    public function hasTag(){
        return true;
    }
    
    
    public function __set($method,callable $function){
        if(is_callable($method)){
            $this->$method = $function;
        }
    }
    

    
}
