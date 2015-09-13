<?php

namespace library\dom\elements;
use library\dom\elements\element;

class elementCollection{
    protected $elements = array();

    public function addElement(element $element){
        if(!in_array($element, $this->elements, true)){
            $this->elements[] = $element;
        }
    }
    public function addElements(array $elements){
        foreach($elements as $element){
            if($element instanceof element){
                $this->addElement($element);
            }
        }
    }
    
    public function getCollection(){
        return $this->elements;
    }
    
    public function index($index){
        return $this->elements[$index];
    }
    
    public function updateElements(array $list){
        $this->elements = array();
        foreach($list as $collection){
            if($collection instanceof element){
                $this->addElement($collection);
            } else {
                $this->addElements($collection->getCollection());
            }
        }
    }
    
    public function __call($name, $arguments) {
        assert(!empty($this->elements));
        $list = array();
        foreach($this->elements as $element){
            if(method_exists($element, $name)){
                //check return value
                /*
                $args = implode(",",$arguments);
                if($arguments[0] instanceof element){
                    $args = $arguments[0];
                }
                $rv = $element->$name($args);
                 * 
                 */
                $rv = call_user_func_array(array($element, $name), $arguments);
                //if return value is instance of elementCollection
                if($rv instanceof elementCollection || $rv instanceof element){
                    //add the return value to the list
                    $list[] = $rv;
                }

            }
        }
        if(!empty($list)){
            $collection  = new elementCollection();
            $collection->addElements($list);
            return $collection;
        }
        return $this;
    }
}