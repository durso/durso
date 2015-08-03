<?php

/**
 * Description of inline
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;
use library\dom\object;
use library\dom\elements\components\text;

class title extends paired{

    
    public function __construct($value) {
        parent::__construct();
        $text = new text($value);
        $this->addComponent($text);
        $this->tag = "title";
        
    }
    public function addComponent(object $component){
        if($this->node->hasChild()){
            $this->removeComponent($this->node->getChild(0));
            $this->node->addChild($component);
        } else {
            parent::addComponent($component);
        }
    }
    public function setValue($value){
        $this->node->getChild(0)->getValue()->setValue($value);
    }


}
