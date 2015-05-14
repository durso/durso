<?php
/**
 * This class is the common superclass of all DOM elements.
 *
 * 
 * @author durso
 */
namespace library\layout\elements;
use library\utils;
use library\event\event;
use library\layout\elements\script;

abstract class element {
    /**
     *
     * @var array All attributes and properties given to an element
     */
    protected $attributes = array();
    /**
     *
     * @var string element inner HTML
     */
    protected $value = "";
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
    /**
     *
     * @var element list of children elements  
     */
    protected $children = null;
    /**
     *
     * @var element parent element  
     */
    protected $parent = null;
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
     * @var boolean javascript response for the element 
     */
    protected $script = false;
    /*
     * 
     * Add CSS class to element
     * @param string $className CSS class name to be added to the element
     * @param string $context (optional) element that will be the context of the Ajax request
     * @return void
     */
    public function addClassName($className,$context = "this"){
        $this->attributes["class"][] = $className;
        if($this->script){
            script::addValue($className,$this->prepareContext($context),"addClass");
        }
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
            if($this->script){
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
    protected function setId($element){
        $this->attributes["id"] = $element."-".utils::randomGenerator(0);
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
     * Get CSS id
     * @return void
     */
    public function hasId(){
        return (isset($this->attributes["id"]) && !empty($this->attributes["id"]));
    }
    public function changeValue($value,$context="this",$method = "html"){
        $this->value = $value;
        if($this->script){
            script::addValue($value,$context,$method);
        }
    }
    public function removeValue(){
        $this->value = "";
    }
    
    public function hasCloseTag(){
        return $this->closeTag;
    }
    /*
     * 
     * Check if element has children
     * @return boolean
     */
    public function hasChildren(){
        return !is_null($this->children);
    }
    /*
     * 
     * Check if element has a parent
     * @return boolean
     */
    public function hasParent(){
        return !is_null($this->parent);
    }
     /*
     * 
     * Add a child to the element
     * @param element $child the object to be added as a child
     * @param string $context (optional) element that will be the context of the Ajax request
     * @param string $method jQuery function to be used to add the child element (Ex: append, prepend)
     * @return void
     */
    public function addChild(element $child, $context="this",$method = "append"){
        $this->children[] = $child;
        $child->setParent($this);
        if($this->script){
            script::addValue($child,$this->prepareContext($context),$method);
        }
    }
     /*
     * 
     * Set the element parent
     * @param element $parent
     * @return void
     */
    public function setParent(element $parent){
        $this->parent = $parent;
    }
     /*
     * 
     * Get the element parent
     * @return element 
     */
    public function getParent(){
       return $this->parent;
    }
     /*
     * 
     * Remove a child from the element
     * @param element $child child element to be removed
     * @param string $context (optional) element that will be the context of the Ajax request
     * @return void
     */
    public function removeChild(element $child, $context = "this"){
        $this->children = utils::array_remove($this->children,$this->child);
        $child->setParent(null);
        if($this->script){
            script::addValue($this->prepareContext($child->getId(),true),$this->prepareContext($context),"remove");
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
     * Return script value
     * @return boolean 
     */
    public function getScript(){
        return $this->script;
    }
    /*
     * Sets script value
     * @param boolean
     * @return void
     * 
     */
    public function setScript($boolean){
        $this->script = $boolean;
    }
    /*
     * Prepares the string that will be the context of the Ajax request
     * @param string $context
     * @return string
     */
    protected function prepareContext($context, $id = false){
        if($id){
            return "'#$context'";
        }
        if($context != "this"){
            if($context == $this->attributes["id"]){
                return "'#".$this->attributes["id"]."'";
            } else {
                return "'.$context'";
            }
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
