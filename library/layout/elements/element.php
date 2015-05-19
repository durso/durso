<?php
/**
 * This class is the common superclass of all DOM elements.
 *
 * 
 * @author durso
 */
namespace library\layout\elements;
use library\tree;
use library\utils;
use library\event\event;
use library\layout\elements\script;

abstract class element extends tree{
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
    /**
     *
     * @var boolean if element has a closing tag
     */
    protected $closeTag = false;
    /*
     * 
     * @var string html to be rendered 
     */
    protected $html = "";
    /*
     * 
     * @var event list of events
     */
    protected $events = array();
    /*
     * 
     * Add CSS class to element
     * @param string $className CSS class name to be added to the element
     * @param string $context (optional) element that will be the context of the Ajax request
     * @return void
     */
    public function addClassName($className,$context = "this"){
        $this->attributes["class"][] = $className;
        if(script::isActive()){
            script::addValue($className,$this->prepareContext($context),"addClass");
        }
    }
    /*
     * 
     * Check if element has class
     * @return void
     */
    public function hasClass(){
        return (isset($this->attributes["class"]) && !empty($this->attributes["class"]));
    }
    /*
     * 
     * Join element attribute array with a string
     * @param array $array list of values to be added to the element
     * @param string $field name of the attribute
     * @return string|void
     */
    protected function renderAttributes(array $array, $field){
        if(count($array) > 0){
            return ' '.$field.'="'.implode(" ",array_unique($array)).'"';
        }
    }
    /*
     * 
     * Create element html attribute
     * @param string $string the value of the attribute
     * @param string $field the name of the attribute
     * @param string $context (optional) element that will be the context of the Ajax request
     * @return string|void
     */
    protected function renderAttribute($string, $field,$context = "this"){
        if($string){
            return ' '.$field.'="'.$string.'"';
            if(script::isActive()){
                script::addKeyValue($field,$string,$this->prepareContext($context),"attr");
            }   
        }
    }
    /*
     * 
     * Set CSS id
     * @param string $element html tag
     * @return void
     */
    public function setId($element,$seed = 0){
        if(!$this->hasId()){
            $this->attributes["id"] = $element."-".utils::randomGenerator($seed);
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
    public function changeValue($value,$context="this",$method = "html"){
        $this->value = $value;
        if(script::isActive()){
            script::addValue($value,$context,$method);
        }
    }
    public function removeValue(){
        $this->value = "";
    }
    
    public function hasCloseTag(){
        return $this->closeTag;
    }
    public function getTag(){
        return $this->tag;
    }
    public function getUid(){
        if($this->hasId()){
            return "#".$this->getId();
        }
        return $this->buildSelector();
    }
     /*
     * 
     * Add a child to the element
     * @param element $child the object to be added as a child
     * @param string $context (optional) element that will be the context of the Ajax request
     * @param string $method (optional) jQuery function to be used to add the child element (Ex: append, prepend)
     * @return void
     */
    public function addChild(element $child, $context="this",$method = "append"){
        parent::addChild($child);
        if(script::isActive()){
            script::addValue($child,$this->prepareContext($context),$method);
        }
    }
     /*
     * 
     * Remove a child from the element
     * @param element $child child element to be removed
     * @param string $context (optional) element that will be the context of the Ajax request
     * @return void
     */
    public function removeChild(element $child){
        parent::removeChild($child);
        if(script::isActive()){
            script::addValue("",$this->prepareContext($child->getUid()),"remove");
        }
    }
    /*
     * 
     * Bind an event and a function to the element
     * @param string $event the name of the event
     * @param callable $callback a function to be bound to the event
     * @return void
     */
    public function bind($event, callable $callback){
        $this->setId($this->tag);
        event::register($this,$event);
        $this->events[$event] = $callback; 
        script::event($this,"click");
    }
    /*
     * Return list of events bound to the element
     * @param string $event
     * @return array
     */
    public function getEvent($event){
        return $this->events[$event];
    }
    /*
     * Prepares the string that will be the context of the Ajax request
     * @param string $context
     * @return string
     */
    protected function prepareContext($context){
        if($context != 'this'){
            return "'$context'";
        }
        return $context;
    }
    /*
     * 
     * Render element to html
     * @return string
     */
    public function render(){
        $this->html .= "<".$this->tag;
        foreach($this->attributes as $key => $value){
            if(is_array($value)){
                $this->html .= $this->renderAttributes($value, $key);
            } else {
                $this->html .= $this->renderAttribute($value, $key);
            }
        }
        $this->html .= ">";
        if(!$this->hasCloseTag()){
            return $this->html;
        }
        $this->html .= $this->value;
        if($this->hasChildren()){
            foreach($this->children as $child){
                $this->html .= $child->render();
            }
        } 
        $this->html .= "</".$this->tag.">";
        return $this->html;
    }
    protected function buildSelector(){
        $selector = "";
        $list = $this->searchAncestorsProperty("hasId");
        foreach($list as $element){
            if($element->hasId()){
                $selector .= "#".$element->getId(); 
            } else {
                $selector .= " > ".$element->getTag();
                $selector .= $element->nthChild();
            }
        }
        $selector .= " > ".$this->getTag()."";
        $selector .= $this->nthChild();
        return $selector;
    }
    public function nthChild(){
        $selector = "";
        if($this->hasSiblings()){
            $index = $this->getSiblingsIndex();
            $index++;
            $selector .= ":nth-child($index)";
        }
        return $selector;
    }

   //abstract public function addListener();
   //abstract public function removeListener();
    public function __call($method, $args){
        if(isset($this->events[$method])){
            call_user_func_array($this->events[$method],$args);
        }
    }
    
    public function __set($method,callable $function){
        if(is_callable($method)){
            $this->$method = $function;
        }
    }
    public function __toString() {
        return $this->render();
    }
    
}
